import { toast } from 'sonner';

export const showToast = {
  success: (message: string) => {
    toast.success(message);
  },
  error: (message: string) => {
    toast.error(message);
  },
  info: (message: string) => {
    toast.info(message);
  },
  warning: (message: string) => {
    toast.warning(message);
  },
  loading: (message: string) => {
    toast.loading(message);
  },
  dismiss: () => {
    toast.dismiss();
  },
};

export default showToast;